# Create directory for data storage
mkdir -p data/chroma_db
mkdir -p data/exports
mkdir -p data/logs

# Create .env file from example
cp .env.example .env

echo "Setup completed! Please edit .env with your API keys."
echo ""
echo "To get started:"
echo "1. Edit .env with your API keys"
echo "2. Install requirements: pip install -r requirements.txt"
echo "3. Process files: python main.py --process-files files/"
echo "4. Query system: python main.py --query 'Your question here'"
