from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# --- Data Loading & Model Training ---

# Load the CSV data
df = pd.read_csv("career_path_in_all_field.csv")

# Prepare features and target;
features = df.drop(columns=["Career", "Field"])
target = df["Career"]

# Split data and train a decision tree classifier
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# --- Knowledge Graph Generation ---
def create_knowledge_graph_image():
    G = nx.Graph()
    
    # Add Field nodes (colored red)
    fields = df["Field"].unique()
    G.add_nodes_from(fields, color="red")
    
    # Add Career nodes (colored blue)
    careers = df["Career"].unique()
    G.add_nodes_from(careers, color="blue")
    
    # Define skill columns 
    skill_columns = df.columns[3:]  # Here, starting from column index 3
    G.add_nodes_from(skill_columns, color="green")
    
    # Create edges: connect each career to its field, and to each skill if the value > 0
    for _, row in df.iterrows():
        G.add_edge(row["Field"], row["Career"])
        for skill in skill_columns:
            if row[skill] > 0:
                G.add_edge(row["Career"], skill)
    
    # Draw the graph and capture as an image in memory
    pos = nx.spring_layout(G, seed=42)
    colors = [G.nodes[node].get("color", "gray") for node in G.nodes()]
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color="gray", font_size=8)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return image_base64

# Generate the knowledge graph image once at startup
knowledge_graph_image = create_knowledge_graph_image()

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        # Extract input features from the form
        input_data = []
        # List of expected feature columns based on your CSV (adjust as necessary)
        feature_cols = [
            "GPA", "Extracurricular_Activities", "Internships", "Projects",
            "Leadership_Positions", "Field_Specific_Courses", "Research_Experience",
            "Coding_Skills", "Communication_Skills", "Problem_Solving_Skills",
            "Teamwork_Skills", "Analytical_Skills", "Presentation_Skills",
            "Networking_Skills", "Industry_Certifications"
        ]
        for col in feature_cols:
            val = request.form.get(col)
            try:
                input_data.append(float(val))
            except (ValueError, TypeError):
                input_data.append(0.0)  # Default to 0 if missing or invalid
        
        # Convert input data to a numpy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)
        prediction = clf.predict(input_array)[0]
    
    return render_template("index.html", prediction=prediction, knowledge_graph_image=knowledge_graph_image)

if __name__ == "__main__":
    app.run(debug=True)
