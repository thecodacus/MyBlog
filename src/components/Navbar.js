import { Link } from "gatsby"
import React from "react"

export default function Navbar() {
	return (
		<nav>
			<h1>The Codacus</h1>
			<div className="links">
				<Link to="/">Home</Link>
				<Link to="/posts">Posts</Link>
				<Link to="/about">About</Link>
			</div>
		</nav>
	)
}
