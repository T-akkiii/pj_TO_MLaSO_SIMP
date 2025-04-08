from pyfreefem import FreeFemRunner

# Loading the script "foo.edp"
runner = FreeFemRunner("foo.edp")

# Loading directly FreeFEM code
script = """mesh Th=square(10,10);"""
runner = FreeFemRunner(script)
runner.execute() 