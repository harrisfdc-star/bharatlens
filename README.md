# Bharat Lens

Local / Streamlit trading dashboard for:
- Zerodha Demat holdings (Kite Connect)
- Nifty 50 buy-the-dip scanner (price vs 50-day MA)

## Run locally

```powershell
cd d:\Cursor_Pro\bharat_lens
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501

## Zerodha setup

1. Create an app at https://developers.kite.trade/ (Personal / free is enough for holdings)
2. Set **Redirect URL** to:
   - Local: `http://localhost:8501/`
   - Streamlit Cloud: `https://YOUR-APP-NAME.streamlit.app/`
3. Enter API Key + API Secret in the **Portfolio** tab and log in

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub
2. Go to https://share.streamlit.io/ and sign in with GitHub
3. Click **New app** → select this repo → Main file path: `app.py`
4. Deploy, then update your Zerodha app Redirect URL to the new Streamlit URL

Do not commit API secrets or `.kite_session.json`.
