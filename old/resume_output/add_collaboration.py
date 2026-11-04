from docx import Document
from docx.text.paragraph import Paragraph
from docx.oxml import OxmlElement
from pathlib import Path
path=Path('resume_output/Mitchell_Lahm_Susquehanna_Resume.docx')
doc=Document(path)
text='Collaborated with Professor Gregory H. Robinson, applying advanced quantum chemistry techniques to provide theoretical support for results from synthetic inorganic experiments.'
assert not any(text == p.text for p in doc.paragraphs)
anchor=next(p for p in doc.paragraphs if p.text.startswith('Collaborated with Matthias'))
e=OxmlElement('w:p'); anchor._p.addnext(e)
p=Paragraph(e,anchor._parent); p.style=doc.styles['List Bullet']; p.add_run(text)
doc.save(path)
source=Path('resume_output/build_resume.py')
if source.exists():
 content=source.read_text(encoding='utf-8-sig')
 target="bullet('Worked with research collaborators"
 if text not in content:
  content=content.replace(target,"bullet("+repr(text)+")\n"+target)
  source.write_text(content,encoding='utf-8')
check=Document(path)
assert sum(p.text==text for p in check.paragraphs)==1
print(text)
