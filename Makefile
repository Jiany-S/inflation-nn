train:
	python src/train.py --lags 36 --features CPI Unemployment OilPrice PPI

backtest:
	python src/backtest.py --lags 36

demo:
	streamlit run app/streamlit_app.py
