"""
FastAPI Todo API — Learning Project (Phases 1-3)

Phase 1: Hello World            -> TODO 4
Phase 2: Path & query params    -> TODO 5, TODO 6
Phase 3: Full CRUD (in-memory)  -> TODO 7, TODO 8, TODO 9

HOW TO RUN:
    1. python -m venv .venv
    2. .venv\\Scripts\\activate          (Windows)
    3. pip install -r requirements.txt
    4. uvicorn main:app --reload
    5. Open http://localhost:8000/docs  <- interactive docs, test everything here

Fill in each TODO below, save, and the server auto-reloads. Test each endpoint
in /docs as you go instead of waiting until the end.
"""

from fastapi import FastAPI, HTTPException
from models import Todo, TodoCreate, TodoUpdate

app = FastAPI(title="Todo API - Python Project")

# ============================================================
# Our "fake database" — just a Python list living in memory.
# Every item will look like: {"id": 1, "title": "...", "completed": False}
# It resets every time the server restarts. That's fine for now — a real
# database comes in a later phase.
# ============================================================
todos: list[dict] = [
    {"id": 1, "title": "Buy milk", "completed": True},
    {"id": 2, "title": "Walk dog", "completed": True},
    {"id": 3, "title": "Study", "completed": False},
]

next_id = 1  # counter used to hand out unique ids


# ============================================================
# TODO 4 — Phase 1: Hello World
# ============================================================
# Create a GET endpoint at "/" that returns {"message": "Todo API is running"}
#
#   @app.get("/")
#   def <function_name>():
#       return {...}
#
# Test it: open http://localhost:8000/ in a browser, or hit it from /docs.
@app.get("/")
def test():
    return {"message": "Todo API is running"}


# ============================================================
# TODO 5 — Phase 2: Path parameter
# ============================================================
# Create a GET endpoint at "/todos/{todo_id}" that:
#   - takes todo_id as an int straight from the URL
#   - loops through `todos` looking for a dict whose "id" matches
#   - returns it if found
#   - otherwise: raise HTTPException(status_code=404, detail="Todo not found")
#
#   @app.get("/todos/{todo_id}")
#   def <function_name>(todo_id: int):
#       for todo in todos:
#           if todo["id"] == todo_id:
#               return todo
#       raise HTTPException(...)


@app.get("/todos/{todo_id}")
def get_todo(todo_id: int):
    for todo in todos:
        if todo["id"] == todo_id:
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")


# ============================================================
# TODO 6 — Phase 2: Query parameter + list endpoint
# ============================================================
# Create a GET endpoint at "/todos" that:
#   - returns all todos by default
#   - accepts an OPTIONAL query param: completed: bool | None = None
#   - if completed is True or False (not None), only return todos matching it
#
#   @app.get("/todos")
#   def <function_name>(completed: bool | None = None):
#       if completed is None:
#           return todos
#       return [t for t in todos if t["completed"] == completed]
#
# Try in browser: /todos            and            /todos?completed=true
#
# NOTE: Put this ABOVE or BELOW TODO 5's route, order between different
# paths ("/todos" vs "/todos/{todo_id}") doesn't matter in FastAPI — only
# matters when two routes could match the SAME url shape.
@app.get("/all_todos")
def get_all_todos(completed: bool | None = None):
    if completed is None:
        return todos
    res = []
    for todo in todos:
        if todo["completed"] == completed:
            res.append(todo)
    return res


# ============================================================
# TODO 7 — Phase 3: Create (POST)
# ============================================================
# Create a POST endpoint at "/todos" that:
#   - accepts a TodoCreate as the request body (just type-hint the param)
#   - builds a new dict: {"id": next_id, "title": ..., "completed": ...}
#     (get title/completed off the TodoCreate object you received, e.g. new_todo.title)
#   - appends it to `todos`
#   - increments next_id (you'll need the `global next_id` statement first)
#   - returns the new todo dict
#
#   @app.post("/todos", response_model=Todo)
#   def <function_name>(new_todo: TodoCreate):
#       global next_id
#       todo = {"id": next_id, "title": new_todo.title, "completed": new_todo.completed}
#       todos.append(todo)
#       next_id += 1
#       return todo


@app.post("/todos", response_model=Todo)
def create_todo(new_todo: TodoCreate):
    global next_id
    next_id = len(todos) + 1
    todo = {"id": next_id, "title": new_todo.title, "completed": new_todo.completed}
    todos.append(todo)
    next_id += 1
    return todo


# ============================================================
# TODO 8 — Phase 3: Update (PUT)
# ============================================================
# Create a PUT endpoint at "/todos/{todo_id}" that:
#   - accepts a TodoUpdate as the request body
#   - finds the matching todo in `todos` (reuse the search logic from TODO 5)
#   - for each field the client actually sent (not None), overwrite it
#     Hint: `if update.title is not None: todo["title"] = update.title`
#   - returns the updated todo
#   - raises 404 if no todo with that id exists
#
#   @app.put("/todos/{todo_id}", response_model=Todo)
#   def <function_name>(todo_id: int, update: TodoUpdate):
#       ...


@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, update: TodoUpdate):
    for todo in todos:
        if todo["id"] == todo_id:
            if update.title is not None:
                todo["title"] = update.title
            if update.completed is not None:
                todo["completed"] = update.completed
            return todo
    raise HTTPException(status_code=404, detail="Todo not found")


# ============================================================
# TODO 9 — Phase 3: Delete (DELETE)
# ============================================================
# Create a DELETE endpoint at "/todos/{todo_id}" that:
#   - finds the todo with that id
#   - removes it from `todos` (todos.remove(todo))
#   - returns {"message": "Todo deleted"}
#   - raises 404 if not found
#
#   @app.delete("/todos/{todo_id}")
#   def <function_name>(todo_id: int):
#       ...


@app.delete("/todos/{todo_id}")
def delete(todo_id: int):
    for todo in todos:
        if todo["id"] == todo_id:
            todos.remove(todo)
            return {"message": "Todo deleted"}
    raise HTTPException(status_code=404, detail="Todo not found")
