from django import forms
from documents.models import String


class DataSetForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super(DataSetForm, self).__init__(*args, **kwargs)
        self.fields['strings'].queryset = String.objects.filter(is_manual=True).exclude(text='')
