"""A spawn target that imports capture/inference only inside the child."""


def worker_entry(spec, output, stop):
    from app.pipeline.worker import worker_main
    worker_main(spec, output, stop)
