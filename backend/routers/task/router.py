from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.db.db_instance import get_db_session
from backend.db.task.dao import (
    get_task_status_from_db,
    get_all_tasks_status_from_db,
    delete_task_from_db,
)
from backend.db.task.models import (
    TasksResult,
    Task,
    TaskType
)
from backend.common.models import (
    Response,
    Result
)
from backend.common.compresser import compress_files

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
    response_model=Task,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Task by Identifier",
    description="Retrieve the status of a specific task using its identifier.",
)
async def get_task(
    identifier: str,
    session: Session = Depends(get_db_session),
) -> Task:
    """
    Retrieve the status of a specific task by its identifier.
    """
    task = get_task_status_from_db(identifier=identifier, session=session)

    if task is not None:
        return task
    else:
        raise HTTPException(status_code=404, detail="Identifier not found")


@task_router.get(
    "file/{identifier}",
    status_code=status.HTTP_200_OK,
    summary="Retrieve Task by Identifier",
    description="Retrieve the status of a specific task using its identifier.",
)
async def get_file_task(
    identifier: str,
    session: Session = Depends(get_db_session),
) -> FileResponse:
    """
    Retrieve the downloadable file response of a specific task by its identifier.
    Compressed by ZIP basically.
    """
    task = get_task_status_from_db(identifier, session)

    if task is not None:
        if task.task_type == TaskType.BGM_SEPARATION:
            output_zip_path = f"{identifier}_bgm_separation.zip"
            output_zip_path = compress_files(
                [task.result.instrumental_path, task.result.vocal_path],
                output_zip_path
            )
            return FileResponse(
                path=output_zip_path,
                status_code=200,
                filename=output_zip_path,
                media_type="application/zip"
            )
        else:
            raise HTTPException(status_code=404, detail=f"File download is only supported for bgm separation."
                                                        f" The given type is {task.task_type}")
    else:
        raise HTTPException(status_code=404, detail="Identifier not found")


@task_router.delete(
    "/{identifier}",
    response_model=Response,
    status_code=status.HTTP_200_OK,
    summary="Delete Task by Identifier",
    description="Delete a task from the system using its identifier.",
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