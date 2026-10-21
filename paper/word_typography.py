"""Pin Chinese Word typography instead of inheriting the template's Japanese locale."""
from lxml import etree
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

BODY_FONT = 'SimSun'
HEADING_FONT = 'SimHei'
WESTERN_FONT = 'Times New Roman'
HEADING_STYLES = {'Title', 'Heading1', 'Heading2', 'TitleChar', 'Heading1Char', 'Heading2Char'}


def pin_run_properties(rpr, east):
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts')
        rpr.insert(0, fonts)
    for key in list(fonts.attrib):
        if 'theme' in etree.QName(key).localname.lower():
            del fonts.attrib[key]
    for slot in ['ascii', 'hAnsi', 'cs']:
        fonts.set(qn('w:' + slot), WESTERN_FONT)
    fonts.set(qn('w:eastAsia'), east)
    fonts.set(qn('w:hint'), 'eastAsia')
    lang = rpr.find(qn('w:lang'))
    if lang is None:
        lang = OxmlElement('w:lang')
        rpr.append(lang)
    lang.set(qn('w:val'), 'en-US')
    lang.set(qn('w:eastAsia'), 'zh-CN')


def normalize_word_fonts(doc):
    defaults = doc.styles.element.find(qn('w:docDefaults')).find(qn('w:rPrDefault')).find(qn('w:rPr'))
    pin_run_properties(defaults, BODY_FONT)
    for style in doc.styles:
        style.element.get_or_add_rPr()
        for rpr in style.element.xpath('.//w:rPr'):
            pin_run_properties(rpr, HEADING_FONT if style.style_id in HEADING_STYLES else BODY_FONT)
    # Direct formatting covers normal text, tables, hyperlinks and page-number fields.
    for part in [doc.part, *[s.footer.part for s in doc.sections]]:
        for paragraph in part.element.xpath('.//w:p'):
            node = paragraph.find(qn('w:pPr'))
            style = node.find(qn('w:pStyle')) if node is not None else None
            east = HEADING_FONT if style is not None and style.get(qn('w:val')) in HEADING_STYLES else BODY_FONT
            for run in paragraph.xpath('.//w:r'):
                pin_run_properties(run.get_or_add_rPr(), east)
    settings = doc.settings.element
    language = settings.find(qn('w:themeFontLang'))
    if language is None:
        language = OxmlElement('w:themeFontLang')
        settings.append(language)
    language.set(qn('w:val'), 'en-US')
    language.set(qn('w:eastAsia'), 'zh-CN')
    theme = doc.part.part_related_by(RT.THEME)
    tree = etree.fromstring(theme.blob)
    ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
    for slot, east in [('majorFont', HEADING_FONT), ('minorFont', BODY_FONT)]:
        node = tree.find('.//a:' + slot, ns)
        node.find('a:latin', ns).set('typeface', WESTERN_FONT)
        node.find('a:ea', ns).set('typeface', east)
        for font in node.findall('a:font', ns):
            if font.get('script') in ['Hans', 'Hant']:
                font.set('typeface', east)
    theme._blob = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
    table = doc.part.part_related_by(RT.FONT_TABLE)
    tree = etree.fromstring(table.blob)
    for name, alternative, family in [(BODY_FONT, '宋体', 'roman'), (HEADING_FONT, '黑体', 'swiss')]:
        font = next((f for f in tree if f.get(qn('w:name')) == name), None)
        if font is None:
            font = OxmlElement('w:font');font.set(qn('w:name'), name);tree.append(font)
        for tag, value in [('altName', alternative), ('charset', '86'), ('family', family)]:
            child = font.find(qn('w:' + tag))
            if child is None:
                child = OxmlElement('w:' + tag);font.append(child)
            child.set(qn('w:val'), value)
    table._blob = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
