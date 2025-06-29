# Surgeon Posture Analysis MVP

A real-time posture analysis application for surgical professionals using MediaPipe Pose and computer vision.

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser:**
   Navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Update for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Deploy the app

## 🔧 Troubleshooting

### OpenCV Import Error

If you encounter OpenCV import errors on Streamlit Cloud:

1. **Check requirements.txt:**
   - Ensure `opencv-python-headless` is used instead of `opencv-python`
   - The headless version is designed for server environments

2. **Test dependencies:**
   ```bash
   python test_deployment.py
   ```

3. **Common solutions:**
   - Clear Streamlit Cloud cache
   - Restart the deployment
   - Check the deployment logs for specific error messages

### MediaPipe Issues

If MediaPipe fails to load:

1. **Check version compatibility:**
   - MediaPipe >= 0.10.9 is required
   - Python 3.8-3.11 is supported

2. **Memory issues:**
   - Reduce model complexity in the code
   - Use `model_complexity=1` instead of `2`

## 📁 Project Structure

```
surgeon-posture-mvp/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── test_deployment.py     # Dependency test script
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.30+
- **Computer Vision**: OpenCV (headless) 4.9+
- **Pose Detection**: MediaPipe Pose v2
- **Data Processing**: NumPy, Pandas
- **Visualization**: Plotly

## 📊 Features

- ✅ Real-time pose detection
- ✅ Ergonomic risk assessment
- ✅ Customizable thresholds
- ✅ Session history tracking
- ✅ Demo mode for testing
- ✅ Responsive web interface

## 🏥 Clinical Applications

- Operating room posture monitoring
- Surgical training and education
- Ergonomic risk assessment
- Injury prevention programs

## ⚠️ Important Notes

- This is a research prototype
- Not intended for clinical diagnosis
- Always consult healthcare professionals for medical advice
- Test thoroughly before clinical use

## 📞 Support

For issues or questions:
- Check the deployment logs
- Run the test script: `python test_deployment.py`
- Review the troubleshooting section above 