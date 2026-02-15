# ♻️ Smart Waste Type Classifier

## 📌 Overview

The Smart Waste Type Classifier is a machine learning-based system that automatically classifies waste images into four categories:

- Plastic
- Paper
- Glass
- Metal

This project uses a Convolutional Neural Network (CNN) trained using supervised learning and provides a user-friendly web interface built with Streamlit.

The system aims to support automated waste segregation and improve recycling efficiency.

---

## 🚀 Features

- CNN-based image classification
- Supervised learning approach
- Automatic image preprocessing
- Streamlit web interface
- Confidence score display
- Color-coded prediction results
- Multi-image upload support (optional enhancement)

---

## 🧠 Machine Learning Details

- Algorithm: Convolutional Neural Network (CNN)
- Learning Type: Supervised Learning
- Image Size: 128 × 128
- Output Classes: 4
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Accuracy: ~75–80% (depends on dataset quality)

---

## 📂 Project Structure

smart_waste_classifier/
│
├── dataset/
│ ├── train/
│ └── test/
│
├── train_model.py
├── app.py
├── waste_model.h5
├── requirements.txt


---

## 🔄 Workflow

1. Images are collected and organized into labeled folders.
2. The CNN model is trained using `train_model.py`.
3. The trained model is saved as `waste_model.h5`.
4. Users upload images through the Streamlit interface.
5. The model predicts the waste type with confidence score.

---

## ⚙️ Installation & Setup
pip install -r requirements.txt


### 1️⃣ Clone the repository


git clone https://github.com/Mayuri21122005/smart-waste-classifier.git
cd smart-waste-classifier

1]Run the application
streamlit run app.py

Open your browser at:
http://localhost:8501

🌍 Real-World Applications

Smart waste bins

Recycling plants

Municipal waste management

Smart city infrastructure

Educational environmental projects


🔮 Future Enhancements

Integration with IoT-based smart dustbins

Real-time camera classification

Mobile application deployment

Cloud deployment

Transfer learning (MobileNet / ResNet) for improved accuracy

Additional waste categories


📊 Limitations

Accuracy depends on dataset size and quality

Lighting and background may affect prediction

Model trained from scratch (no transfer learning)

🏆 Conclusion

This project demonstrates how deep learning can be applied to solve real-world environmental challenges through intelligent automation and image classification.


## 📸 Application Preview
<img width="1920" height="1020" alt="Screenshot 2026-02-15 114236" src="https://github.com/user-attachments/assets/0abc8bd9-d7d7-41cf-8552-faea2ee5627a" />
<img width="1920" height="1020" alt="Screenshot 2026-02-15 114410" src="https://github.com/user-attachments/assets/ad1d9368-a12c-4a55-a4e1-a70744e36130" />
<img width="1920" height="1020" alt="Screenshot 2026-02-15 114431" src="https://github.com/user-attachments/assets/2605a98b-deb8-4631-a163-52bf5b09ff82" />
<img width="1920" height="1020" alt="Screenshot 2026-02-15 114503" src="https://github.com/user-attachments/assets/eff6ff8a-d2dd-4f7f-8915-f4bf4c9381d9" />


