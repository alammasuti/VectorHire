"""
Pydantic models — these define the "shape" of data going in and out of your API.

Think of these like TypeScript interfaces, but with a big difference: FastAPI
actually CHECKS incoming data against them at runtime. A TS interface disappears
the moment your code compiles — a Pydantic model is still there, validating,
when a real request comes in.
"""

from pydantic import BaseModel


class TodoCreate(BaseModel):
    """
    What the CLIENT sends when creating a new todo (POST /todos body).

    TODO 1: Add two fields:
      - title: str                  (required — every todo needs a title)
      - completed: bool = False     (optional — defaults to False if not sent)

    Example of the syntax (delete this comment once you've added the real fields):
        title: str
        completed: bool = False
    """

    title: str
    completed:bool = False


class TodoUpdate(BaseModel):
    """
    What the CLIENT sends when updating a todo (PUT /todos/{id} body).

    TODO 2: Add the same two fields as TodoCreate, but make BOTH optional
    this time, using `| None = None`. Why? So the client can send just
    {"completed": true} without being forced to resend the title too.

        title: str | None = None
        completed: bool | None = None
    """

    title: str| None = None
    completed: bool| None = None


class Todo(TodoCreate):
    """
    What the API RETURNS to the client — a TodoCreate plus a server-generated id.

    Notice this class inherits from TodoCreate (the parentheses above), so it
    automatically already has `title` and `completed`. You only need to add
    what's NEW here.

    TODO 3: Add one field:
      - id: int
    """
    
    id: int
