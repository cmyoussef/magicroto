class UIElementDocs:

    @classmethod
    def get_doc(cls, ui_name):
        if hasattr(cls, ui_name):
            return getattr(cls, ui_name).replace('    ', '')
        return ''
