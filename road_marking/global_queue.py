import multiprocessing

manager = multiprocessing.Manager()
task_queue = manager.Queue()
cut_queue = manager.Queue()

remote_process_queue = manager.Queue()
remote_cut_queue = manager.Queue()

divide_queue = manager.Queue()
extend_queue = manager.Queue()

remote_extend_queue = manager.Queue()

online_queue = manager.Queue()
online_result = manager.list()
