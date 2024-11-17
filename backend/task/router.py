from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db.db_instance import get_db_session
from ..db.task.dao import (
    get_task_status_from_db,
    get_all_tasks_status_from_db,
    delete_task_from_db,
)
from ..db.task.models import (
    TasksResult,
    Task
)
from ..common.models import (
    Response,
    Result
)

task_router = APIRouter(prefix="/task", tags=["Tasks"])


@task_router.get(
    "/all",
    response_model=TasksResult,
    status_code=status.HTTP_200_OK,
    summary="Retrieve All Task Statuses",
    description="Retrieve the statuses of all tasks available in the system.",
)
async def get_all_tasks_status(
    session: Session = Depends(get_db_session),
) -> TasksResult:
    """
    Retrieve all tasks.
    """
    return get_all_tasks_status_from_db(session)


@task_router.get(
    "/{identifier}",
    status_code=status.HTTP_200_OK,
    summary="Retrieve Task by Identifier",
    description="Retrieve the status of a specific task using its identifier.",
    response_model=Task,
)
async def get_task(
    identifier: str,
    session: Session = Depends(get_db_session),
) -> Task:
    """
    Retrieve the status of a specific task by its identifier.
    """
    task = get_task_status_from_db(identifier, session)

    if task is not None:
        return task
    else:
        raise HTTPException(status_code=404, detail="Identifier not found")


@task_router.delete(
    "/{identifier}",
    status_code=status.HTTP_200_OK,
    summary="Delete Task by Identifier",
    description="Delete a task from the system using its identifier.",
    response_model=Response,
)
async def delete_task(
    identifier: str,
    session: Session = Depends(get_db_session),
) -> Response:
    """
    Delete a task by its identifier.
    """
    if delete_task_from_db(identifier, session):
        return Response(identifier=identifier, message="Task deleted")
    else:
        raise HTTPException(status_code=404, detail="Task not found")
