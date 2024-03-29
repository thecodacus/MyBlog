import React from "react"
import Navbar from "./Navbar"
import "../styles/global.scss"
export default function Layout({ children }) {
	return (
		<div className="layout">
			<Navbar />
			<div className="content">{children}</div>
			<footer>
				<p>Copyright 2021 The Codacus</p>
			</footer>
		</div>
	)
}
