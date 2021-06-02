.PHONY: clean

clean:
	cargo clean
	rm -r Cargo.lock gaussians.egg-info gaussians/*.so gaussians/__pycache__
