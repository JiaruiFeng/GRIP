from .base import BaseEvalDataset


class GRIPEvalDataset(BaseEvalDataset):
    template = (
        "Given the context graph titled {title}, please answer the following question: {question} Response in "
        "the following format:")

    def __getitem__(self, idx):
        question = self.questions[idx]
        if isinstance(question, list):
            question, roots = question
        answer = self.answers[idx]

        assert self.title is not None, "Graph title is required for GRIP evaluation."
        user_content = self.template.format(question=question, title=self.title) + self.answer_format
        sample = self._format_chat_template(user_content)
        return self.tokenizer(sample, return_tensors="pt")["input_ids"], question, answer
