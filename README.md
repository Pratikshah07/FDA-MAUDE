# MAUDE Data Processor

A comprehensive web application for processing and analyzing FDA MAUDE (Manufacturer and User Facility Device Experience) medical device adverse event data. Features AI-powered IMDRF code mapping, manufacturer normalization, secure authentication, and data visualization capabilities.

## 🚀 Features

- **🔐 Secure Authentication** - User registration, login, and password recovery via email
- **📊 Data Processing** - Clean and standardize MAUDE CSV/Excel files with intelligent column detection
- **🏷️ IMDRF Code Mapping** - Hierarchical mapping of device problems to IMDRF codes using Annex A-G structure
- **🏭 Manufacturer Normalization** - AI-assisted manufacturer name cleanup and M&A resolution with web verification
- **📅 Date Standardization** - Automatic conversion to DD-MM-YYYY format
- **✅ Data Validation** - Comprehensive validation with regulatory compliance checks
- **🎨 Modern UI** - Beautiful, responsive web interface with drag-and-drop file upload
- **📈 Data Visualization** - Interactive dashboards and analytics (coming soon)

## 🛠️ Tech Stack

- **Backend:** Python 3.8+, Flask
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **AI/ML:** Groq API (Llama 3.1 70B)
- **Data Processing:** Pandas, OpenPyXL, xlrd
- **Authentication:** Flask-Login, bcrypt
- **Email:** Flask-Mail (Gmail/Mailgun)

## 📖 Usage

1. **Register/Login** - Create an account or sign in
2. **Upload MAUDE File** - Drag and drop or select CSV/Excel file
3. **Upload IMDRF Annexure** (Optional) - Upload Annexes A-G file for IMDRF code mapping
4. **Process** - Click "Process File" to start the pipeline
5. **Download** - Download the cleaned and enriched output file

## 🔄 Data Processing Pipeline

The application processes MAUDE data through the following stages:

1. **Column Identification** - AI-assisted detection of required columns
2. **Data Cleaning** - Removal of specified columns and data sanitization
3. **Date Standardization** - Conversion to DD-MM-YYYY format
4. **Row Filtering** - Removal of rows with missing critical dates
5. **Manufacturer Normalization** - AI-powered name cleanup and M&A resolution
6. **IMDRF Mapping** - Hierarchical mapping to IMDRF codes (Level-1 → Level-2 → Level-3)
7. **Validation** - Comprehensive output validation with regulatory compliance checks

## 🔒 Security

- Passwords hashed with bcrypt
- Password reset tokens expire after 1 hour
- User data stored securely in JSON files
- API keys managed via environment variables
- No sensitive data committed to repository

## 🚧 Future Enhancements

- Interactive data visualization dashboards
- Advanced analytics and reporting
- Export to multiple formats (PDF, JSON)
- Batch processing capabilities
- API endpoints for programmatic access
- Real-time processing status updates

## 📊 Streamlit Signal Dashboard (new)

A simple Streamlit dashboard has been added to visualize event counts over time with threshold detection.

How to run locally:

1. Install the new dependencies (or the full set):

   pip install -r requirements.txt

2. Run the Streamlit app:

   streamlit run streamlit_app.py

Notes:

- The Streamlit app expects a *cleaned* MAUDE file (Excel or CSV) with the following columns present (exact headers): `Event Type`, `Manufacturer`, `Device Problem`, `IMDRF Code`, and either `Event Date` or `Date Received` (dates must be in `DD-MM-YYYY` format).
- The Streamlit app is independent of the Flask app and intentionally does NOT call any LLMs.
- If you prefer a different filename for the dashboard, rename `streamlit_app.py` and run with `streamlit run <filename>`.


## 👤 Author

Devarsh Radadia

## 🙏 Acknowledgments

- FDA MAUDE database
- IMDRF (International Medical Device Regulators Forum)
- Groq for AI/ML capabilities

---

**Note:** This is a production-ready application designed for processing FDA medical device adverse event data with regulatory compliance in mind.
