
# 🕳️ Pothole Detection and Cost Estimation System

A computer vision-based system for detecting potholes from road images or video feeds. This project identifies potholes, calculates their surface area, and estimates the cost for repairs based on predefined metrics.

## 🚀 Features

- 📷 Real-time pothole detection using OpenCV and pre-trained models
- 📐 Surface area estimation of potholes from images
- 💸 Automatic repair cost estimation using area and cost-per-sq.ft logic
- 📊 Optional visualization and annotation of potholes
- 🧪 Supports image or video input for testing

## 📁 Project Structure

```
pothole_Detection/
├── model/                    # Model files or weights (if applicable)
├── data/                     # Sample images or videos for testing
├── utils/                    # Utility functions for area, visualization, etc.
├── main.py                   # Entry-point to run detection
├── area_calculator.py        # Calculates pothole area
├── cost_estimator.py         # Computes estimated repair cost
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation
```

## 🛠️ Technologies Used

- Python 3.x
- OpenCV
- NumPy
- Matplotlib (for visualization)
- Pre-trained ML or DL model (YOLO / Haar cascade / Custom CNN) *(optional based on your setup)*

## ⚙️ How It Works

1. **Input**: User provides an image or video frame of a road.
2. **Detection**: The algorithm detects pothole regions.
3. **Area Calculation**: Area of the pothole is calculated based on pixel-to-cm² ratio.
4. **Cost Estimation**: Estimated cost is calculated using:
   ```
   Estimated Cost = Area (in sq.ft) × Cost per sq.ft (₹)
   ```

## 🧪 Getting Started

### ✅ Prerequisites

- Python 3.8+
- pip

### 📦 Installation

```bash
git clone https://github.com/sridharreddy7831/pothole_Detection.git
cd pothole_Detection
pip install -r requirements.txt
```

## 🖼️ Usage

### 💡 To run detection and cost estimation on a sample image:

```bash
python main.py --image data/road1.jpg
```

### 💡 For video input:

```bash
python main.py --video data/road_video.mp4
```

### ⚙️ Parameters

- `--image`: Path to the input image file
- `--video`: Path to the input video file
- `--model`: (Optional) Path to custom detection model
- `--show`: Display annotated output

## 🧮 Cost Estimation Logic

By default:

- **Conversion**: 1 pixel ≈ 0.25 cm² *(adjustable)*
- **Rate**: ₹50 per square foot *(can be modified in `cost_estimator.py`)*

## 📌 Future Enhancements

- Integration with GPS for pothole location tagging
- Real-time mobile/web app interface
- Training on custom datasets for better accuracy
- Support for drone/vehicle video feeds

## 🙌 Contributing

Feel free to fork this project and submit pull requests. For major changes, open an issue first to discuss what you’d like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✍️ Author

**Gangireddy Sumanth Reddy**  
📧 [sumanthr447@gmail.com](mailto:sumanthr447@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/sridharreddy7831)

## 📷 Sample Output

*(Add sample before/after image or detection result screenshots here)*
