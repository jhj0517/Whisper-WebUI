from typing import Dict, Any
from sqlalchemy.orm import Session
from fastapi import Depends

from ..db_instance import handle_database_errors, get_db_session
from .models import Task, TasksResult, TaskStatus


@handle_database_errors
def add_task_to_db(
    session,
    status=TaskStatus.QUEUED,
    task_type=None,
    language=None,
    task_params=None,
    file_name=None,
    url=None,
    audio_duration=None,
):
    """
    Add task to the db
    """
    task = Task(
        status=status,
        language=language,
        file_name=file_name,
        url=url,
        task_type=task_type,
        task_params=task_params,
        audio_duration=audio_duration,
    )
    session.add(task)
    session.commit()
    return task.uuid


@handle_database_errors
def update_task_status_in_db(
    identifier: str,
    update_data: Dict[str, Any],
    session: Session,
):
    """
    Update task status and attributes in the database.

    Args:
        identifier (str): Identifier of the task to be updated.
        update_data (Dict[str, Any]): Dictionary containing the attributes to update along with their new values.
        session (Session, optional): Database session. Defaults to Depends(get_db_session).

    Returns:
        None
    """
    task = session.query(Task).filter_by(uuid=identifier).first()
    if task:
        for key, value in update_data.items():
            setattr(task, key, value)
        session.commit()


@handle_database_errors
def get_task_status_from_db(
    identifier: str, session: Session
):
    """Retrieve task status from db"""
    task = session.query(Task).filter(Task.uuid == identifier).first()
    if task:
        return task
    else:
        return None


@handle_database_errors
def get_all_tasks_status_from_db(session: Session):
    """Get all tasks from db"""
    columns = [Task.uuid, Task.status, Task.task_type]
    query = session.query(*columns)
    tasks = [task for task in query]
    return TasksResult(tasks=tasks)


@handle_database_errors
def delete_task_from_db(identifier: str, session: Session):
    """Delete task from db"""
    task = session.query(Task).filter(Task.uuid == identifier).first()

    if task:
        # If the task exists, delete it from the database
        session.delete(task)
        session.commit()
        return True
    else:
        # If the task does not exist, return False
        return False
