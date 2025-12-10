            else:
                # ================= DL PIPELINE =================
                model, meta = load_dl_model()
                if model is None or meta is None:
                    return

                feature_cols = meta["feature_cols"]
                scaler = meta["scaler"]
                window = meta["window"]

                # 1) Get raw features in the same order as training
                try:
                    X_raw = df_feat[feature_cols].values
                except KeyError:
                    missing_cols = [c for c in feature_cols if c not in df_feat.columns]
                    st.error(
                        "Feature mismatch for DL model.\n\n"
                        f"Missing features in app: `{missing_cols}`.\n"
                        "Make sure engineer_features_for_app() matches train_model_dl.py."
                    )
                    return

                # 2) Scale using training scaler
                X_scaled = scaler.transform(X_raw)

                # 3) Build sequences for ALL available windows
                X_seq = build_sequences_for_app(X_scaled, window=window)
                if X_seq.shape[0] == 0:
                    st.error(
                        f"Not enough data to form a sequence of length {window} for LSTM."
                    )
                    return

                # 4) Predict for all sequences
                dir_probs, mag_preds = model.predict(X_seq)
                dir_probs = dir_probs.ravel()
                mag_preds = mag_preds.ravel()
                dir_preds = (dir_probs >= 0.5).astype(int)

                # 5) Create DL prediction columns explicitly
                df_feat_dl = df_feat.copy()
                df_feat_dl["dl_pred_up"] = np.nan
                df_feat_dl["dl_prob_up"] = np.nan
                df_feat_dl["dl_pred_return"] = np.nan

                # First 'window' rows have no prediction; align sequences to rows [window:]
                df_feat_dl.iloc[window:, df_feat_dl.columns.get_loc("dl_pred_up")] = dir_preds
                df_feat_dl.iloc[window:, df_feat_dl.columns.get_loc("dl_prob_up")] = dir_probs
                df_feat_dl.iloc[window:, df_feat_dl.columns.get_loc("dl_pred_return")] = mag_preds

                # 6) Keep only rows where we actually have predictions
                valid_mask = (
                    df_feat_dl["dl_prob_up"].notna()
                    & df_feat_dl["dl_pred_return"].notna()
                )
                df_valid = df_feat_dl[valid_mask]

                if df_valid.empty:
                    st.error("No valid DL predictions found after alignment.")
                    return

                # 7) Use last row with predictions
                latest_row = df_valid.iloc[-1]
                latest_index = latest_row.name
                latest_close = float(latest_row["Close"])
                latest_rsi = float(latest_row["rsi_14"])
                latest_prob_up = float(latest_row["dl_prob_up"])
                latest_pred_return = float(latest_row["dl_pred_return"])
                direction_label = "UP" if latest_prob_up >= 0.5 else "DOWN / FLAT"

                closes_valid = df_valid["Close"].values
                preds_valid = df_valid["dl_pred_up"].values
                pattern = get_pattern_label(preds_valid, closes_valid)

                # --- SUMMARY TAB (DL) ---
                with tab_summary:
                    st.subheader(f"Market pattern for `{ticker}` (DAILY – LSTM)")

                    if "UPTREND" in pattern:
                        st.markdown(f"### 🟢 {pattern}")
                    elif "DOWNTREND" in pattern:
                        st.markdown(f"### 🔴 {pattern}")
                    elif "SIDEWAYS" in pattern:
                        st.markdown(f"### 🟡 {pattern}")
                    else:
                        st.markdown(f"### {pattern}")

                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    mcol1.metric("Last close", f"${latest_close:,.2f}")
                    mcol2.metric("Prob. of UP", f"{latest_prob_up:.1%}")
                    mcol3.metric("RSI (14)", f"{latest_rsi:.1f}")
                    mcol4.metric("Predicted move", f"{latest_pred_return * 100:+.2f}%")

                    st.markdown("### Latest bar prediction (LSTM)")
                    st.write(f"- **Bar time:** `{latest_index}`")
                    st.write(f"- **Model:** LSTM (Deep Learning, DAILY)")
                    st.write(f"- **Model expectation:** **{direction_label}** for the **next day**")
                    st.write(f"- **Probability of UP:** `{latest_prob_up:.2%}`")
                    st.write(
                        f"- **Predicted price change:** `{latest_pred_return * 100:+.2f}%` (next day)"
                    )

                # --- CHARTS TAB (same as RF) ---
                with tab_charts:
                    st.subheader("Price & moving averages")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_feat.index, df_feat["Close"], label="Close")
                    ax.plot(df_feat.index, df_feat["ma_5"], label="MA 5")
                    ax.plot(df_feat.index, df_feat["ma_10"], label="MA 10")
                    ax.plot(df_feat.index, df_feat["ma_20"], label="MA 20")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Price")
                    ax.legend()
                    st.pyplot(fig)

                    st.subheader("RSI (14)")
                    fig2, ax2 = plt.subplots(figsize=(10, 2.5))
                    ax2.plot(df_feat.index, df_feat["rsi_14"], label="RSI 14")
                    ax2.axhline(70, linestyle="--")
                    ax2.axhline(30, linestyle="--")
                    ax2.set_ylim(0, 100)
                    ax2.set_ylabel("RSI")
                    st.pyplot(fig2)

                # --- RECENT SIGNALS TAB (DL) ---
                with tab_signals:
                    st.subheader("Recent LSTM outputs (last 15 bars with predictions)")
                    recent = (
                        df_valid[
                            ["Close", "rsi_14", "dl_pred_up", "dl_prob_up", "dl_pred_return"]
                        ]
                        .tail(15)
                        .rename(
                            columns={
                                "Close": "Close price",
                                "rsi_14": "RSI (14)",
                                "dl_pred_up": "Pred. up? (1=yes)",
                                "dl_prob_up": "Prob. of up",
                                "dl_pred_return": "Pred. return (decimal)",
                            }
                        )
                    )
                    st.dataframe(recent)

                # --- MODEL DETAILS TAB (DL) ---
                with tab_model:
                    st.subheader("LSTM (Deep Learning) model details")
                    st.markdown(
                        f"""
**Mode:** `DAILY`  
**Horizon:** `next day`  

Architecture:
- 2× LSTM layers (64 → 32 units)
- Dropout (0.3)
- Two heads:
  - `direction` (sigmoid) → UP/DOWN
  - `magnitude` (linear) → next_return (decimal)

Loss:
- Binary cross-entropy (direction)
- MSE (magnitude)
- Joint training with loss weights (1.0, 0.5)

Features (same as RandomForest):
- Returns: return_1d, return_2d, return_5d
- MAs: ma_5, ma_10, ma_20
- MA ratios: ma_5_20_ratio, ma_10_20_ratio
- Volatility: vol_5, vol_10
- RSI(14)

Input to LSTM:
- Sliding window of length `{window}` bars,
  scaled with StandardScaler fitted during training.
"""
                    )
