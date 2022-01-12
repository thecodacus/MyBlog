import { styled } from "@stitches/react"

export default styled("div", {
	display: "flex",
	flexDirection: "column",
	padding: "4rem",
	backgroundColor: "#fafafa",
	// fontFamily: '"Roboto", sans-serif',

	fontSize: "1rem",
	lineHeight: "1.5",
	borderRadius: "0.5rem",
	boxShadow: "0 0.25rem 0.5rem rgba(0, 0, 0, 0.1)",
	alignContent: "center",
	// alignItems: "center",
	// justifyContent: "center",
	// textAlign: "center",

	"*": {
		color: "#333",
	},
	"& h1": {
		fontSize: "2rem",
		fontWeight: "400",
		margin: "0.5rem 0",
	},
	"& h2": {
		fontSize: "1.5rem",
		fontWeight: "400",
		margin: "0.5rem 0",
	},
	"& h3": {
		fontSize: "1.25rem",
		fontWeight: "400",
		margin: "0.5rem 0",
	},
	"& h4": {
		fontSize: "1rem",
		fontWeight: "400",
		margin: "0.5rem 0",
	},
	"& h5": {
		fontSize: "0.875rem",
		fontWeight: "400",
		margin: "0.5rem 0",
	},
	"& h6": {
		fontSize: "0.75rem",
		fontWeight: "400",
		margin: "0.5rem 0",
	},
	"& p": {
		margin: "0.5rem 0",
	},
	"& a": {
		color: "#0070f3",
		textDecoration: "none",
		"&:hover": {
			textDecoration: "underline",
		},
	},
	"& ul": {
		margin: "0.5rem 0 1rem",
		padding: "0 0 0 1.5rem",
		"& li": {
			margin: "0.25rem 0",
			padding: "0",
		},
	},
	"& ol": {
		margin: "0.5rem 0 1rem",
		padding: "0 0 0 1.5rem",
		"& li": {
			margin: "0.25rem 0",
			padding: "0",
		},
	},
	"& blockquote": {
		margin: "0.5rem 0 1rem",
		padding: "0 0 0 1.5rem",
		borderLeft: "0.25rem solid #0070f3",
		"& p": {
			margin: "0",
		},
	},
	"& img": {
		maxWidth: "100%",
	},
	"& pre": {
		margin: "0.5rem 0",
		padding: "0.5rem",
		overflow: "auto",
		fontFamily: "monospace",
		backgroundColor: "#fafafa",
		borderRadius: "0.25rem",
		"& code": {
			fontSize: "inherit",
			padding: "0",
			color: "#333",
		},
	},
	"& table": {
		width: "100%",
		margin: "0.5rem 0",
		borderCollapse: "collapse",
		"& th": {
			fontWeight: "600",
			borderBottom: "1px solid #ddd",
		},
		"& td": {
			borderBottom: "1px solid #ddd",
		},
	},
	"& hr": {
		margin: "1rem 0",
		border: "0",
		borderTop: "1px solid #ddd",
	},
})
