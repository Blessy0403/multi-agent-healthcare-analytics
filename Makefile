.PHONY: run dashboard install clean

install:
	pip install -r requirements.txt

run:
	python main.py

dashboard:
	streamlit run dashboard/app.py

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
