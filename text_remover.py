import pikepdf
import re

def empty_pdf(input_path, input_path2):
    with pikepdf.open(input_path) as pdf:
        for page in pdf.pages:
            if '/Contents' in page:
                contents = page.Contents
                # Convert the page stream(s) to bytes
                streams = []
                if isinstance(contents, pikepdf.Array):
                    for obj in contents:
                        streams.append(bytes(obj.read_bytes()))
                else:
                    streams.append(bytes(contents.read_bytes()))
                
                # Combine into one big stream
                stream_data = b'\n'.join(streams)

                # Remove text operators between BT and ET
                new_stream = re.sub(rb'BT.*?ET', b'', stream_data, flags=re.S)

                # Replace page stream with cleaned version
                page.Contents = pdf.make_stream(new_stream)
        

        pdf.save(input_path2)
