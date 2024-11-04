# myapp/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def remove_extension(value):
    """
    Remove file extension from filename.
    """
    if '.' in value:
        return value.rsplit('.', 1)[0]
    return value
