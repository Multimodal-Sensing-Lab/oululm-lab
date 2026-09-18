"""Report the active Python stack and optionally test one CUDA forward/backward."""
import argparse
import importlib.metadata
import json
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output')
    args = parser.parse_args()
    import torch
    report = {'python': sys.version, 'executable': sys.executable, 'architecture': platform.machine(),
              'pytorch_cuda_build': torch.version.cuda, 'versions': {}}
    for package in ['torch','transformers','peft','accelerate','datasets','pyyaml','scikit-learn','bitsandbytes']:
        try:
            report['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['versions'][package] = None
    report['optimizer_resume_supported'] = tuple(int(i) for i in torch.__version__.split('+')[0].split('.')[:2]) >= (2,6)
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise SystemExit('CUDA is unavailable in this process; compare with the normal desktop terminal')
        report['gpu'] = torch.cuda.get_device_name(0)
        report['capability'] = torch.cuda.get_device_capability(0)
        report['free_bytes'], report['total_bytes'] = torch.cuda.mem_get_info()
    x = torch.randn(32, 32, device=args.device, requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    report['forward_backward'] = args.device + ': passed'
    print(json.dumps(report, indent=2))
    if args.output:
        from .io import write_json
        write_json(args.output, report)


if __name__ == '__main__':
    main()
