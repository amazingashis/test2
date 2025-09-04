# Create directory for data storage
mkdir -p data/chroma_db
mkdir -p data/exports
mkdir -p data/logs

echo "Setup completed!"
echo ""
echo "To get started:"
echo "1. Set your Databricks API key: export DATABRICKS_API_KEY=your_key"
echo "2. Install requirements: pip install -r requirements.txt"
echo "3. Process files: python main.py --process-files files/"
echo "4. Query system: python main.py --query 'Your question here'"
