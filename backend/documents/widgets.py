from django.forms import widgets
from django.utils.safestring import mark_safe
from django import forms
from .models import *


class ImageWidget(forms.widgets.ClearableFileInput):
    template_name = "image_preview_widget.html"

class ImageWithRectanglesWidget(forms.widgets.ClearableFileInput):
    def __init__(self, page=None, *args, **kwargs):
        self.page = page
        super().__init__(*args, **kwargs)

    def render(self, name, value, attrs=None, renderer=None):
        if self.page and self.page.image:
            image_html = f'''
            <div style="position: relative; display: inline-block;" id="image-container">
                <img id="image" src="{self.page.image.url}" style="max-width: 100%; display: block;">
            '''
            for string in self.page.string_set.all():
                style = (
                    f'position: absolute; '
                    f'border: 2px solid red; '
                    f'color: red; '
                    f'display: flex; '
                    f'align-items: center; '
                    f'justify-content: center; '
                    f'overflow: hidden; '
                    f'text-align: center;'
                )
                image_html += f'<div class="rectangle" data-x1="{string.x1}" data-y1="{string.y1}" data-x2="{string.x2}" data-y2="{string.y2}" style="{style}"><span class="rectangle-text">{string.text}</span></div>'
            image_html += '</div>'

            script = '''
            <script>
                function getOptimalFontSize(text, containerWidth, containerHeight, context) {
                    const initialFontSize = 10;
                    context.font = initialFontSize + "px Arial";
                    const textWidth = context.measureText(text).width;
                    const optimalFontSize = Math.min(containerWidth / textWidth * initialFontSize, containerHeight);
                    return optimalFontSize;
                }

                function updateRectangles() {
                    var image = document.getElementById("image");
                    var rects = document.querySelectorAll(".rectangle");

                    var naturalWidth = image.naturalWidth;
                    var naturalHeight = image.naturalHeight;

                    var displayWidth = image.clientWidth;
                    var displayHeight = image.clientHeight;

                    var widthRatio = displayWidth / naturalWidth;
                    var heightRatio = displayHeight / naturalHeight;

                    var canvas = document.createElement('canvas');
                    var context = canvas.getContext('2d');

                    rects.forEach(function(rect) {
                        var x1 = parseInt(rect.getAttribute("data-x1"));
                        var y1 = parseInt(rect.getAttribute("data-y1"));
                        var x2 = parseInt(rect.getAttribute("data-x2"));
                        var y2 = parseInt(rect.getAttribute("data-y2"));

                        var rectWidth = (x2 - x1) * widthRatio;
                        var rectHeight = (y2 - y1) * heightRatio;

                        rect.style.left = (x1 * widthRatio) + "px";
                        rect.style.top = (y1 * heightRatio) + "px";
                        rect.style.width = rectWidth + "px";
                        rect.style.height = rectHeight + "px";

                        var text = rect.querySelector(".rectangle-text");
                        if (text) {
                            var optimalFontSize = getOptimalFontSize(text.innerText, rectWidth, rectHeight, context);
                            text.style.fontSize = optimalFontSize + "px";
                            text.style.whiteSpace = "nowrap";
                        }
                    });
                }

                window.onload = updateRectangles;
                window.onresize = updateRectangles;

                document.addEventListener("DOMContentLoaded", function() {
                    var rects = document.querySelectorAll(".rectangle");
                    rects.forEach(function(rect) {
                        rect.addEventListener("mouseenter", function() {
                            var text = rect.querySelector(".rectangle-text");
                            if (text) {
                                text.style.display = "none";
                            }
                        });
                        rect.addEventListener("mouseleave", function() {
                            var text = rect.querySelector(".rectangle-text");
                            if (text) {
                                text.style.display = "block";
                            }
                        });
                    });
                });
            </script>
            '''
            image_html += script
            return mark_safe(image_html)
        else:
            input_html = super().render(name, value, attrs, renderer)
            return mark_safe(input_html)
    