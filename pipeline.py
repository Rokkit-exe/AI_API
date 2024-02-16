

class Pipeline:
    def __init__(self, model_name, model_path, framework, tokenizer, model_config):
        self.model = model_name
        self.model_path = model_path
        self.framework = framework
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.tasks = []

    def task(self, depends_on=None):
        def inner(f):
            self.tasks.append((f, depends_on))
            return f
        return inner

    def run(self):
        for f, depends_on in self.tasks:
            if depends_on:
                depends_on()
            f()