import ast
import sys 
import os
package_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.getenv('NUKE_DEV', False) and package_path not in sys.path:
    sys.path.insert(0, package_path)

from nukebridge.executors.core.baseexecutor import BaseExecutor, logger

from magicroto.core import rvm


class RVMExecutor(BaseExecutor):

    def __init__(self):
        super().__init__()
        self.converter = rvm.Converter(self.args.variant, self.args.checkpoint, self.args.device)

    def setup_parser(self):
        super().setup_parser()
        self.parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
        self.parser.add_argument('--checkpoint', type=str, required=True)
        self.parser.add_argument('--device', type=str, default='cuda')
        # self.parser.add_argument('--input', type=str, required=True)
        self.parser.add_argument('--downsample-ratio', type=float, default=0)
        self.parser.add_argument('--output-type', type=str, choices=['png'], default='png')
        self.parser.add_argument('--seq-chunk', type=int, default=1)
        self.parser.add_argument('--num-workers', type=int, default=0)
        self.parser.add_argument('--disable-progress', action='store_true')

    def run(self):
        logger.info(f'running rvm.converter.convert {self.args_dict}')

        self.converter.convert(
            input_path=self.args.input,
            ouput_path=self.args.output,
            downsample_ratio=self.args.downsample_ratio,
            output_type=self.args.output_type,
            seq_chunk=self.args.seq_chunk,
            frame_range=self.frame_range,
            num_workers=self.args.num_workers,
            progress=not self.args.disable_progress
        )


if __name__ == '__main__':
    executor = RVMExecutor()
    try:
        lvl = int(executor.args_dict.get('logger_level'))
    except TypeError:
        lvl = 20
    logger.setLevel(lvl)
    executor.run()
