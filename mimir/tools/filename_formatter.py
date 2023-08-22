import re
from jinja2 import Environment


class FileNameFormatter:
    def __init__(self, template_string, delimiters={"field": "_", "word": "-"}, extension=".pdf"):
        env = Environment()
        env.filters['upper'] = lambda x: x.upper()
        env.filters['lower'] = lambda x: x.lower()
        env.filters['replace_spaces'] = lambda x, d: x.replace(" ", d)

        self.template = env.from_string(template_string)
        self.delimiters = delimiters
        self.extension = extension

    def generate_filename(self, data):
        # Render the template with the provided data and delimiters
        filename = self.template.render(data=data, sd=self.delimiters["word"], mwd=self.delimiters["field"])

        # Replace any non-word characters with the word delimiter
        filename = re.sub(r'\W+', self.delimiters["word"], filename).rstrip(self.delimiters["field"])

        # Append the file extension
        filename += self.extension

        return filename


if __name__ == "__main__":
    # Template string
    template_parts = [
        "{{ data.author|upper|replace_spaces(sd) }}",
        "{{ mwd }}",
        "{{ data.year }}",
        "{{ mwd }}",
        "{{ data.title|replace_spaces(sd) }}",
        "{{ extension }}"
    ]

    template_string = "".join(template_parts)

    # Custom delimiters
    delimiters = {"field": "__", "word": "-"}

    # Create an instance of FileNameFormatter
    formatter = FileNameFormatter(template_string, delimiters)

    # Example usage
    data = {
        "author": "Smith, J.",
        "title": "A Study on Gravity",
        "year": "2021"
    }

    filename = formatter.generate_filename(data)
    print(filename)
