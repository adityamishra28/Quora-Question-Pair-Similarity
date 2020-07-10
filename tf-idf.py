Python 3.6.1 (v3.6.1:69c0db5050, Mar 21 2017, 01:21:04) 
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "copyright", "credits" or "license()" for more information.
>>> WARNING: The version of Tcl/Tk (8.5.9) in use may be unstable.
Visit http://www.python.org/download/mac/tcltk/ for current information.
from flask import Flask, render_template,request,redirect,url_for,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://aditya@localhost:5432/todoo'
db = SQLAlchemy(app)
migrate = Migrate(app,db)

class Todo(db.Model):
  __tablename__ = 'todos'
  id = db.Column(db.Integer, primary_key=True)
  description = db.Column(db.String(), nullable=False)
  completed = db.Column(db.Boolean, nullable=False, default = False)

  def __repr__(self):
    return f'<Todo {self.id} {self.description}>'


@app.route('/todos/create', methods=['POST'])
def create_todo():
  description = request.get_json()['description']
  todo = Todo(description=description)
  db.session.add(todo)
  db.session.commit()
  return jsonify({
    'description': todo.description
  })

@app.route('/todos/<todo_id>/set-completed', methods=['POST'])
def set_completed_todo(todo_id):
  try:
    completed = request.get_json()['completed']
    print('completed', completed)
    todo = Todo.query.get(todo_id)
    todo.completed = completed
    db.session.commit()
  except:
    db.session.rollback()
  finally:
    db.session.close()
  return redirect(url_for('index'))

@app.route('/todos/<todo_id>', methods=['DELETE'])
def delete_todo(todo_id):
  try:
    Todo.query.filter_by(id=todo_id).delete()
    db.session.commit()
  except:
    db.session.rollback()
  finally:
    db.session.close()
  return jsonify({ 'success': True })

@app.route('/')
def index():
  return render_template('index.html', data=Todo.query.order_by('id').all())
SyntaxError: multiple statements found while compiling a single statement
from flask import Flask, render_template,request,redirect,url_for,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://aditya@localhost:5432/todoo'
db = SQLAlchemy(app)
migrate = Migrate(app,db)

class Todo(db.Model):
  __tablename__ = 'todos'
  id = db.Column(db.Integer, primary_key=True)
  description = db.Column(db.String(), nullable=False)
  completed = db.Column(db.Boolean, nullable=False, default = False)

  def __repr__(self):
    return f'<Todo {self.id} {self.description}>'


@app.route('/todos/create', methods=['POST'])
def create_todo():
  description = request.get_json()['description']
  todo = Todo(description=description)
  db.session.add(todo)
  db.session.commit()
  return jsonify({
    'description': todo.description
  })

@app.route('/todos/<todo_id>/set-completed', methods=['POST'])
def set_completed_todo(todo_id):
  try:
    completed = request.get_json()['completed']
    print('completed', completed)
    todo = Todo.query.get(todo_id)
    todo.completed = completed
    db.session.commit()
  except:
    db.session.rollback()
  finally:
    db.session.close()
  return redirect(url_for('index'))

@app.route('/todos/<todo_id>', methods=['DELETE'])
def delete_todo(todo_id):
  try:
    Todo.query.filter_by(id=todo_id).delete()
    db.session.commit()
  except:
    db.session.rollback()
  finally:
    db.session.close()
  return jsonify({ 'success': True })

@app.route('/')
def index():
  return render_template('index.html', data=Todo.query.order_by('id').all())
>>> 
>>> 
>>> 
>>> 
>>> <html>
  <head>
    <title>Todo App</title>
    <style>
      #error {
        display: none;
      }

      ul{
        list-style: none;
        padding: 0;
        margin: 0;
        width: 200px;
      }
      li{
        clear:both;
      }
      li button{
        -webkit-appearance: none;
        border: none;
        outline: none;
        color: red;
        float: right;
        cursor: pointer;
        font-size: 20px;
      }
    </style>
  </head>
  <body>
    <div id="error" class="hidden">Something went wrong!</div>
    <form id="form" method="post" action="/todos/create">
      <input type="text" id="description" name="description" />
      <input type="submit" value="Create" />
    </form>
    <ul id="todos">
      {% for d in data %}
      <li>
        <input class="check-completed" data-id="{{d.id}}" type="checkbox" {% if d.completed %} checked  {%endif%}/> 
        {{ d.description }}
        <button class = "delete-button" data-id="{{d.id}}">&cross;</button>
      </li>
      {% endfor %}
    </ul>
    <script>
      const deleteBns = document.querySelectorAll('.delete-button')
      for(let i=0;i<deleteBns.length;i++) {
        const btn= deleteBns[i];
        btn.onclick = function(e){
          const todoId = e.target.dataset['id'];
          fetch('/todos/'+ todoId, {
            method: 'DELETE'
          })
          .then(function() {
            const item = e.target.parentElement;
            item.remove();
          })
        }
      }
      const checkboxes = document.querySelectorAll('.check-completed');
      for(let i=0;i<checkboxes.length; i++){
        const checkbox = checkboxes[i];
        checkboxes.onchange = function(e){
          console.log('event', e);
          const newCompleted = e.target.checked;
          const todoId = e.target.dataset['id'];
          fetch('/todos/' + todoId + 'set-completed', {
            method : "POST",
            body: JSON.stringify({
              'completed': newCompleted
            }),
            headers: {
              'Content-Type': 'application/json'
            }
          })
          .catch(function() {
            document.getElementById('error').className = '';
        })
        }
      }
      const descInput = document.getElementById('description');
      document.getElementById('form').onsubmit = function(e) {
        e.preventDefault();
        const desc = descInput.value;
        descInput.value = '';
        fetch('/todos/create', {
          method: 'POST',
          body: JSON.stringify({
            'description': desc,
          }),
          headers: {
            'Content-Type': 'application/json',
          }
        })
        .then(response => response.json())
        .then(jsonResponse => {
          console.log('response', jsonResponse);
          li = document.createElement('li');
          li.innerText = desc;
          document.getElementById('todos').appendChild(li);
          document.getElementById('error').className = 'hidden';
        })
        .catch(function() {
          document.getElementById('error').className = '';
        })
      }
    </script>
  </body>
</html>
