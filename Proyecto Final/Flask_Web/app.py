from flask import Flask, render_template, request, redirect
import pandas as pd
import model

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        url = request.form['url']
        
        # Call the recommendation code with the URL input and retrieve the dataframe
        dataframe = model.pipeline(url)
        
        # Pass the dataframe and loading message to the template for rendering
        return render_template('recommendations.html', recommendations = dataframe.to_html(), loading_message = None)
    
    return render_template('index.html', loading_message=None)

@app.route('/new_recommendation', methods=['GET'])
def new_recommendation():
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
