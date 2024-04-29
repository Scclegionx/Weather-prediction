from django import template

register = template.Library()

@register.filter(name='translate_day')
def translate_day(value):
    # Here you can implement the logic to translate the day
    # For example:
    translation_dict = {
        'Monday': 'Thứ Hai',
        'Tuesday': 'Thứ Ba',
        'Wednesday': 'Thứ Tư',
        'Thursday': 'Thứ Năm',
        'Friday': 'Thứ Sáu',
        'Saturday': 'Thứ Bảy',
        'Sunday': 'Chủ Nhật',
    }
    return translation_dict.get(value, value)  # Return translated day or original value if not found
