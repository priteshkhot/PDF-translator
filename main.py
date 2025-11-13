import fitz 
from googletrans import Translator
import asyncio
from testtest import empty_pdf

# Async because you have to wait till translaton is complete to insert translated text
async def translate_pdf_in_place(input_path, input_path2, output_path, target_lang="hi", font_path="fonts/NotoSansDevanagari-Regular.ttf"):
    # Load PDF
    
    doc = fitz.open(input_path)
    doc2 = fitz.open(input_path2)
    font_alias = "NotoSansDevanagari" # Register the font with a clean alias name (no spaces) - (YES, blank spaces in font name was an issue)
    font = fitz.Font(fontfile=font_path) # this just preloads the font, its not used but removing it throws an error
    translator = Translator()
   

    for page_num, page in enumerate(doc, start=1):
        print(f"[Info] Translating page {page_num}/{len(doc)}...")

        # Extract text blocks: (x0, y0, x1, y1, text, block_no, line_no, word_no)
        blocks = page.get_text("blocks")

        # Load pages for doc2
        paged2 = doc2[page_num - 1]
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()
            if not text:
                continue

            try:
                translation = await translator.translate(text, src='en', dest=target_lang)
                translated_text = translation.text
            except Exception as e:
                print(f"[Warning] Translation failed: {e}")
                translated_text = text

            # Insert translated text in the same location
            rect = fitz.Rect(x0, y0, x1, y1)
            paged2.insert_font(font_alias, fontfile=font_path)
            # Use the alias instead of font.name
            paged2.insert_textbox(
                rect,
                translated_text,
                fontname=font_alias,
                fontsize=10,
                color=(0, 0, 0),
                fontfile=font_path
            )
            
    print("[Info] PDF successfully translated")

    # Save new translated PDF
    doc2.save(output_path, garbage=4, deflate=True, clean=True) #remove cache to compress the file
    doc2.close()
    doc.close()
    print(f"[Info] Translated PDF saved at: {output_path}")


if __name__ == "__main__":
    input_pdf = 'input/fecu101.pdf' # don't use '\' (trust me)
    language = "hi"
    pdf_name= input_pdf.split(sep="/")[-1]
    
    input_pdf2 = f'raw/textless_{pdf_name}.pdf'
    output_pdf = f'output/GT_{pdf_name}'
    font_path = "fonts/NotoSansDevanagari-Regular.ttf"
    

    empty_pdf(input_pdf, input_pdf2)
    asyncio.run(translate_pdf_in_place(input_pdf, input_pdf2, output_pdf, target_lang=language, font_path=font_path))


