from django import forms
import .models Region

class RegionForm(forms.ModelForm):

    class Meta:
        model = Region
        fields =('region')

    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['region'].queryset=Region.Objects.None()