from docx import Document
from docx.shared import Inches, Pt, RGBColor
from pathlib import Path
out=Path(r'C:\Users\Mitchell\Desktop\concordantmodes\resume_output');out.mkdir(exist_ok=True)
d=Document(); s=d.sections[0];s.top_margin=Inches(.55);s.bottom_margin=Inches(.55);s.left_margin=Inches(.65);s.right_margin=Inches(.65)
normal=d.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(10.5);normal.paragraph_format.space_after=Pt(4)
normal.paragraph_format.line_spacing=1.0
for name in ['Title','Heading 1','Heading 2']:
 st=d.styles[name];st.font.name='Calibri';st.font.color.rgb=RGBColor(0,0,0)
d.styles['Title'].font.size=Pt(23);d.styles['Title'].paragraph_format.space_after=Pt(3)
d.styles['Heading 1'].font.size=Pt(11);d.styles['Heading 1'].paragraph_format.space_before=Pt(9);d.styles['Heading 1'].paragraph_format.space_after=Pt(4)
def p(t,b=False):
 q=d.add_paragraph();r=q.add_run(t);r.bold=b;return q
def bullet(t):
 q=d.add_paragraph(style='List Bullet');q.paragraph_format.space_after=Pt(3);q.add_run(t)
def h(t):d.add_paragraph(t,'Heading 1')
d.add_paragraph('Mitchell E Lahm','Title')
p('Athens, GA | 970-417-0077 | Mitchell.Lahm@uga.edu')
p('github.com/MitchLahm | linkedin.com/in/mitchell-lahm-86808510a')
h('Professional Summary')
p('Computational scientist with a PhD in computational quantum chemistry, open-source scientific software experience, and a record of collaborative research. Work spans molecular simulation methods, computational benchmarks, and technical publications. Experienced in explaining complex methods through advanced instruction, research presentations, and student supervision.')
h('Core Expertise')
p('Scientific software • Computational method development • Benchmark design and analysis • Quantum chemistry • Technical writing • Research collaboration • Technical presentations • Research mentoring')
h('Research and Software Experience')
p('Research Scientist | University of Georgia | Appointed 2022',True)
bullet('Conduct computational quantum chemistry research spanning molecular vibrations, reaction energetics, and chemical bonding, with findings published in peer-reviewed journals.')
bullet('Collaborate with researchers at the University of Georgia and with Matthias F. Bickelhaupt and Wesley D. Allen on computational studies and benchmark research.')
bullet('Contribute to benchmark studies assessing density functional theory against high-level reference calculations, including pericyclic reactions and hydrogen bonding.')
bullet('Participate as co-principal investigator in an NSF grant submission with Brandon Rotavera in 2025.')
p('Open Source Scientific Software | concordantmodes',True)
p('github.com/CCQC/concordantmodes')
bullet('Contribute to open-source software for the Concordant Mode Approach to molecular vibrational analysis, connecting computational research with reusable scientific tools.')
bullet('Published research on the Concordant Mode Approach in JACS (2022), its convergent CMA-2 extension in JCTC (2024), and intermolecular benchmarks in JPCA (2026).')
h('Teaching and Technical Communication')
p('Advanced Quantum Chemistry Instructor | University of Georgia | 2026',True)
bullet('Teach advanced quantum chemistry; previously delivered theory lectures in 2023 and supported the course through teaching assistantships.')
bullet('Supervised undergraduate researchers, including a 2019 project resulting in a 2020 Journal of Organic Chemistry publication.')
bullet('Presented research at PsiCon (2021, 2025), ICQC (2023), and CBOND (2024); serve as a referee for The Journal of Chemical Physics.')
h('Education')
p('PhD in Computational Quantum Chemistry | University of Georgia | 2021',True)
p('Advisor: Henry F. Schaefer III')
p('BS in Chemistry | Bethel University | 2017',True)
h('Selected Recognition and Service')
p('Outstanding Teaching Assistant Award, 2019 • Co-organizer, CCQC undergraduate summer research program, 2018 • Scientific outreach presentations at Clarke Middle School')
d.core_properties.author='Mitchell E. Lahm';d.core_properties.title='Mitchell Lahm Resume'
d.save(out/'Mitchell_Lahm_Resume.docx')
