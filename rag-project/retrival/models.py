from django.db import models

# Create your models here.


class UploadedPDF(models.Model):
    """
    Model to store details about uploaded PDFs.
    """
    file = models.FileField(upload_to='uploads/',
                            help_text="PDF file uploaded by the user")
    uploaded_at = models.DateTimeField(
        auto_now_add=True, help_text="Timestamp when the PDF was uploaded")
    name = models.CharField(max_length=255, blank=True,
                            null=True, help_text="Name of the uploaded PDF")

    def __str__(self):
        return self.name if self.name else f"PDF uploaded at {self.uploaded_at}"


class Query(models.Model):
    """
    Model to store user queries and responses.
    """
    user_query = models.TextField(help_text="The query asked by the user")
    response = models.TextField(
        help_text="The response provided by the system", blank=True, null=True)
    created_at = models.DateTimeField(
        auto_now_add=True, help_text="Timestamp when the query was made")
    pdf = models.ForeignKey(
        UploadedPDF, on_delete=models.CASCADE, related_name='queries', help_text="PDF related to this query"
    )

    def __str__(self):
        return f"Query on {self.pdf.name} at {self.created_at}"
