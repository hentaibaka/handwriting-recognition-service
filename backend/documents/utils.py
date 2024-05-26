def handle_page_img(instance, filename: str) -> str:
    filename = f"{instance.document.name}_{instance.page_num}.{filename.split('.')[1]}"
    return f"images/{instance.document.name}/{filename}"
