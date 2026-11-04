from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
p=r'C:\Users\Mitchell\Desktop\concordantmodes\resume_output\Mitchell_Lahm_Resume.docx'
d=Document(p)
def link(par,label,url):
 h=OxmlElement('w:hyperlink');h.set(qn('r:id'),par.part.relate_to(url,RT.HYPERLINK,is_external=True))
 r=OxmlElement('w:r');pr=OxmlElement('w:rPr');color=OxmlElement('w:color');color.set(qn('w:val'),'0563C1');pr.append(color);u=OxmlElement('w:u');u.set(qn('w:val'),'single');pr.append(u);r.append(pr);t=OxmlElement('w:t');t.text=label;r.append(t);h.append(r);par._p.append(h)
for par in d.paragraphs:
 if '970-417-0077' in par.text:
  par.clear();par.add_run('Athens, GA | 970-417-0077 | ');link(par,'mitchlahm@gmail.com','mailto:mitchlahm@gmail.com')
 elif 'github.com/MitchLahm' in par.text:
  par.clear();link(par,'github','https://github.com/MitchLahm');par.add_run(' | ');link(par,'LinkedIn','https://www.linkedin.com/in/mitchell-lahm-86808510a')
d.save(p)
from zipfile import ZipFile
with ZipFile(p) as z:
 xml=z.read('word/document.xml').decode(); rels=z.read('word/_rels/document.xml.rels').decode()
 for v in ['mitchlahm@gmail.com','>github<','>LinkedIn<']:assert v in xml,v
 for v in ['mailto:mitchlahm@gmail.com','https://github.com/MitchLahm','https://www.linkedin.com/in/mitchell-lahm-86808510a']:assert v in rels,v
 assert 'Mitchell.Lahm@uga.edu' not in xml
print('Contact text and all three hyperlink targets verified.')
