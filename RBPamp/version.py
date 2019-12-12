name="RBPamp"
version="0.9.20"
license="MIT"
authors=["Marvin Jens", ]

def git_commit():
    import subprocess
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    gc_name = os.path.join(path, "../git_commit")
    if os.path.exists(gc_name):
         return open(gc_name, 'r').read()
    else:
        try:
            git = subprocess.Popen(["git","describe","--always"], cwd=path, stdout=subprocess.PIPE).communicate()[0].rstrip()
        except:
            git = "unknown"
        return git
