from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

import os
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()

# ==============================
# PostgreSQL SETUP
# ==============================

DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "employees_db")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")


def get_connection():
    """Create a new DB connection."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def init_db():
    """Create employees table if it doesn't exist."""
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS employees (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        age INT NOT NULL,
                        department VARCHAR(100) NOT NULL,
                        salary NUMERIC(12, 2) NOT NULL
                    );
                    """
                )
    finally:
        conn.close()


init_db()

# ==============================
# TOOLS
# ==============================


@tool()
def add_employee(name: str, age: int, department: str, salary: float):
    """
    Add a new employee to the database.

    Args:
        name: Full name of the employee
        age: Age of the employee (integer)
        department: Department name
        salary: Current salary (float)
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO employees (name, age, department, salary)
                    VALUES (%s, %s, %s, %s)
                    RETURNING *;
                    """,
                    (name, age, department, salary),
                )
                employee = cur.fetchone()
                return dict(employee)
    except Exception as e:
        return f"Error adding employee: {e}"
    finally:
        conn.close()


@tool()
def list_employees():
    """
    Return a list of all employees in the database.
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM employees ORDER BY id;")
                employees = cur.fetchall()
                return [dict(emp) for emp in employees]
    except Exception as e:
        return f"Error listing employees: {e}"
    finally:
        conn.close()


@tool()
def get_employee_by_id(employee_id: int):
    """
    Fetch a single employee's details by their ID.
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM employees WHERE id = %s;",
                    (employee_id,),
                )
                employee = cur.fetchone()
                if employee:
                    return dict(employee)
                return f"No employee found with id {employee_id}"
    except Exception as e:
        return f"Error fetching employee: {e}"
    finally:
        conn.close()


@tool()
def update_employee_salary(employee_id: int, new_salary: float):
    """
    Update the salary of an employee.

    Args:
        employee_id: ID of the employee whose salary should be updated
        new_salary: New salary value
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE employees
                    SET salary = %s
                    WHERE id = %s
                    RETURNING *;
                    """,
                    (new_salary, employee_id),
                )
                employee = cur.fetchone()
                if employee:
                    return dict(employee)
                return f"No employee found with id {employee_id}"
    except Exception as e:
        return f"Error updating salary: {e}"
    finally:
        conn.close()


@tool()
def delete_employee(employee_id: int):
    """
    Delete an employee from the database by ID.
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "DELETE FROM employees WHERE id = %s RETURNING id;",
                    (employee_id,),
                )
                deleted = cur.fetchone()
                if deleted:
                    return f"Employee with id {employee_id} deleted."
                return f"No employee found with id {employee_id}"
    except Exception as e:
        return f"Error deleting employee: {e}"
    finally:
        conn.close()


tools = [
    add_employee,
    list_employees,
    get_employee_by_id,
    update_employee_salary,
    delete_employee,
]

# ==============================
# LANGGRAPH STATE + GRAPH
# ==============================


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = init_chat_model(model_provider="openai", model="gpt-4.1")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}


tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()

# ==============================
# MAIN LOOP
# ==============================


def main():
    print("Employee DB Chatbot (PostgreSQL + LangGraph)")
    print("Try things like:")
    print('- "Add a new employee John Doe, 30 years old, in DevOps with salary 90000"')
    print('- "Show me all employees"')
    print('- "Update salary of employee 1 to 120000"')
    print('- "Delete employee with id 2"')
    print()

    while True:
        user_query = input("> ")

        state: State = {
            "messages": [{"role": "user", "content": user_query}]
        }

        for event in graph.stream(state, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
