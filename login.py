from flask import Flask, render_template, redirect,request, url_for

app = Flask(__name__)

app.secret_key = 'your-secret-key-here'



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # You can add any logic here, but for now, we simply redirect
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/home')
def home():
    # Render the base.html page
    return render_template('base.html')

if __name__ == '__main__':
    app.run(debug=True)
