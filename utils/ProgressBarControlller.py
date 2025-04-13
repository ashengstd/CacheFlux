import rich.progress


class ProgressController:
    def __init__(self, enable: bool = True):
        self.enable = enable

    def __enter__(self):
        if self.enable:
            self.progress = rich.progress.Progress()
            self.progress.start()

    def add_task(self, description: str, total: float | None):
        if self.enable:
            return self.progress.add_task(description=description, total=total)

    def update_task(self, task_id: int, advance: float = 1.0):
        if self.enable:
            self.progress.update(task_id=task_id, advance=advance)

    def remove_task(self, task_id: int):
        if self.enable:
            self.progress.remove_task(task_id=task_id)
