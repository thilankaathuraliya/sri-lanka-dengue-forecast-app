# Sri Lankan Dengue Forecast — MSc demonstration

## 1. Export artifacts from Colab
Run Section 10 of `Sri_Lanka_Dengue_Improved_CatBoost_Colab.ipynb`. Download and copy into this app directory:

- `dengue_catboost_improved.cbm` (REQUIRED; trained model)
- `dengue_model_metadata.json` (REQUIRED; exact feature order and model configuration)
- `dengue_city_month_predictions.csv` (optional; historical evaluation chart)

The dataset `data/dengue_history.xlsx` is included from the workbook supplied for the assignment. Confirm you have permission to redistribute the dataset before publishing a public repository. If not, keep it in a private repository and follow the data owner's terms.

## 2. Test locally
Install Python 3.10+ and run:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 3. Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `data/dengue_history.xlsx`, and the exported model + metadata. Include the optional evaluation CSV if desired.
2. Go to https://share.streamlit.io/ and select **Create app** / **Deploy an app**.
3. Select your GitHub repository, branch, and `app.py` as the entrypoint.
4. Deploy and test the location selector, historical chart, model prediction, SHAP chart, and metrics.
5. Capture screenshots for your MSc report and record a 3–5 minute demo.

## Important methodological notes
- This is a **historical one-step-ahead January 2021 forecast** using recorded data through December 2020, not a 2026 forecast.
- The saved CatBoost model was trained through December 2018. It was evaluated diagnostically on 2019–2020 data already inspected during model development.
- Historical evaluation: revised CatBoost MAE 105.995, RMSE 242.278, R² 0.704; previous-month baseline MAE 97.037, RMSE 237.311, R² 0.716.
- Do not use as a real-time public-health or clinical decision support tool.
