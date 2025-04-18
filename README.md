# Career Guidance App

The **Career Guidance App** is a Flask-based web application that helps users explore career paths based on their skills, education, and interests. It uses machine learning to predict suitable careers and visualizes relationships between fields, careers, and skills through an interactive knowledge graph.

---

## Features

- **Career Prediction**: Predicts the most suitable career based on user input (e.g., GPA, skills, extracurricular activities).
- **Knowledge Graph**: Visualizes relationships between career fields, job roles, and required skills.
- **Interactive Web Interface**: A user-friendly interface built with Flask for easy interaction.

---

## Project Structure

Career-guidance-app/
│
├── app.py                     # Main Flask application
├── career_path_in_all_field.csv # Dataset containing career-related data
├── careermapping.ipynb         # Jupyter Notebook for data exploration and preprocessing
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # HTML template for the web interface
├── static/
│   └── (optional static files like CSS/JS)
└── .gitignore                  # Git ignore file

```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Career-guidance-app.git
   cd Career-guidance-app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Fill in the form with your details (e.g., GPA, skills) and submit to get career predictions.

---

## Dataset

The application uses the career_path_in_all_field.csv dataset, which contains information about:
- Career fields
- Job roles
- Skills and qualifications (e.g., GPA, extracurricular activities, internships)

---

## Knowledge Graph

The app generates a knowledge graph that visualizes:
- **Nodes**: Career fields, job roles, and skills.
- **Edges**: Relationships between fields, careers, and required skills.

The graph is generated using `networkx` and rendered as a static image.

---

## Model

- **Algorithm**: Decision Tree Classifier
- **Training**: The model is trained on the dataset to predict careers based on user input.
- **Future Improvements**:
  - Use a more robust model (e.g., Random Forest or Gradient Boosting).
  - Implement hyperparameter tuning for better accuracy.

---

## Development

### Running Jupyter Notebook
To explore or preprocess the dataset, open the careermapping.ipynb file:
```bash
jupyter notebook careermapping.ipynb
```

### Adding Features
- Update app.py to include new features or improve the prediction model.
- Modify `index.html` in the templates folder to enhance the user interface.

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- **Libraries Used**:
  - Flask
  - pandas
  - scikit-learn
  - matplotlib
  - networkx
- **Dataset**: career_path_in_all_field.csv
