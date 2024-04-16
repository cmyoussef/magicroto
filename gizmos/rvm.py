import nuke
from nukebridge.gizmos.core.base import Base

from magicroto.config.config_loader import ConfigLoader


class RVM(Base):

    def __init__(self, gizmo=None, name=None):
        self.config_loader = ConfigLoader()
        self._MODULE_NAME = 'magicroto'
        super().__init__(gizmo=gizmo, name=name)
        self.copy_alpha_nods()


    def create_generate_knobs(self):
        self.create_generate_tab()
        # variant menu
        if not self.gizmo.knob('variant_menu'):
            logger_level_menu = nuke.Enumeration_Knob('variant_menu', 'Variant', ['mobilenetv3', 'resnet50'])
            logger_level_menu.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(logger_level_menu)

        format_mult = self.gizmo.knob(f'down_size_mult_knob')
        if not format_mult:
            format_mult = nuke.Double_Knob(f'down_size_mult_knob', f'Down_Size')
            format_mult.setFlag(nuke.STARTLINE)
            format_mult.setFlag(nuke.DISABLED)
            format_mult.setValue(1)
            self.gizmo.addKnob(format_mult)

        if not self.gizmo.knob('auto_down_size_knob'):
            auto_down_size_knob = nuke.Boolean_Knob('auto_down_size_knob', 'Auto Down Size')
            auto_down_size_knob.setValue(True)
            auto_down_size_knob.clearFlag(nuke.STARTLINE)
            self.gizmo.addKnob(auto_down_size_knob)

        if not self.gizmo.knob('chunk_size_knob'):
            chunk_size_knob = nuke.Double_Knob('chunk_size_knob', 'Chunk Size')
            chunk_size_knob.setValue(5)
            chunk_size_knob.setRange(1, 50)
            self.gizmo.addKnob(chunk_size_knob)

        if not self.gizmo.knob('device_menu'):
            device_level_menu = nuke.Enumeration_Knob('device_menu', 'Device', ['GPU', 'CPU'])
            device_level_menu.setFlag(nuke.STARTLINE)
            self.gizmo.addKnob(device_level_menu)

        super().create_generate_knobs()

    def update_args(self):
        super().update_args()
        variant = self.gizmo.knob('variant_menu').value()
        self.args['variant'] = variant
        self.args['checkpoint'] = f'{self.cache_dir}/rvm_{variant}.pth'
        self.args['frame_range'] = f'{self.frame_range}'
        self.args['seq-chunk'] = int(self.gizmo.knob('chunk_size_knob').value())


    def knobChanged(self, knob=None):
        knob = knob or nuke.thisKnob()
        check_ = super().knobChanged(knob)
        if not check_:
            return False

        if knob.name() == 'chunk_size_knob':
            knob.setValue(int(knob.value()))

        device = self.gizmo.knob('device_menu').value()
        if device == 'GPU':
            device = 'cuda'
        else:
            device = 'cpu'
        self.args['device'] = device

        auto_down_check = self.gizmo.knob('auto_down_size_knob').value()
        if auto_down_check == True:
            self.args['downsample-ratio'] = 0
        elif auto_down_check == False:
            down_size = self.gizmo.knob(f'down_size_mult_knob').value()
            if down_size <= .9:
                self.args['downsample-ratio'] = down_size

        if knob.name() == 'auto_down_size_knob':
            isEnabled = knob.value()
            if isEnabled:
                self.gizmo.knob('down_size_mult_knob').setFlag(nuke.DISABLED)
            else:
                self.gizmo.knob('down_size_mult_knob').clearFlag(nuke.DISABLED)

    def copy_alpha_nods(self):
        # Clear selection
        nuke.selectAll()
        nuke.invertSelection()

        # Assuming self.get_node("Read1", 'Read') returns the Read node
        read_node = self.get_node("Read1", 'Read')

        # 2. Create an Expression Node
        expression_node = self.get_node("alpha_convert", 'Expression')
        expression_node.setInput(0, read_node)
        expression_node['expr3'].setValue("clamp(r+g+b, 0, 1)")

        # 3. Create a Copy Node
        copy_node = self.get_node("copy_alpha", 'Copy')
        copy_node.setInput(0, self.input_node)  # Existing node connected to the output
        copy_node.setInput(1, expression_node)
        copy_node['from0'].setValue("alpha")
        copy_node['to0'].setValue("alpha")

        # Update the output node connection to use the Copy node
        self.output_node.setInput(0, copy_node)
