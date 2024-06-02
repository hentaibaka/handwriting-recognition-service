from django import forms
from .widgets import *


class PDFUploadForm(forms.Form):
    pdf_file = forms.FileField(label="Выберите PDF файл")

class PageForm(forms.ModelForm):
    class Meta:
        model = Page
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            self.fields['image'].widget = ImageWithRectanglesWidget(self.instance)
 
class StringForm(forms.ModelForm):
    class Meta:
        model = String
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            self.fields['image_preview'] = forms.CharField(
                required=False,
                widget=forms.widgets.TextInput(attrs={'readonly': 'readonly', 'blank': True}),
                initial=self.instance.page.image.url
            )
