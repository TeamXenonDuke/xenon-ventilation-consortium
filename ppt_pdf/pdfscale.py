from PyPDF2 import PdfFileWriter, PdfFileReader

with open("report_clinical_005027.pdf", "rb") as in_f1:
    with open("report_technical_005027.pdf",'rb') as in_f2:
        input1 = PdfFileReader(in_f1)
        input2 = PdfFileReader(in_f2)
        output = PdfFileWriter()

        p = input1.getPage(0)
        p.scale(4,4) # scale it up by a factor of 2
        output.addPage(p)

        p = input2.getPage(0)
        p.scale(4,4) # scale it up by a factor of 2
        output.addPage(p)

        with open("report_005027_big.pdf", "wb") as out_f:
            output.write(out_f)

# with open("report_technical_005027.pdf", "rb") as in_f:
#     input1 = PdfFileReader(in_f)
#     output = PdfFileWriter()
#
#     p = input1.getPage(0)
#     p.scale(4,4) # scale it up by a factor of 2
#     output.addPage(p)
#
#     with open("report_technical_005027_big.pdf", "wb") as out_f:
#         output.write(out_f)
