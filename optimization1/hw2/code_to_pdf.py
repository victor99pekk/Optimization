import pygments
from pygments import lexers
from pygments.formatters import HtmlFormatter
import pdfkit

def code_to_html(code):
    lexer = lexers.get_lexer_by_name("python")
    formatter = HtmlFormatter(full=True, style="colorful")
    html_code = pygments.highlight(code, lexer, formatter)
    return html_code

def html_to_pdf(html_code, output_file):
    pdfkit.from_string(html_code, output_file)

def main():
    # Read the Python code from a file
    input_file = 'plot.py'
    output_file = 'scatter.pdf'
    
    with open(input_file, 'r') as file:
        code = file.read()
    
    # Convert code to HTML
    html_code = code_to_html(code)
    
    # Convert HTML to PDF
    html_to_pdf(html_code, output_file)
    
    print(f"PDF generated: {output_file}")

if __name__ == "__main__":
    main()